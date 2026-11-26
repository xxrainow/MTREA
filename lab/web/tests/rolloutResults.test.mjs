import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import ts from 'typescript';

const source = await readFile(new URL('../src/rollouts/results.ts', import.meta.url), 'utf8');
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023 } });
const { groupRolloutResults, outcomeCounts, resultTargetSelection, formatElapsed } =
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

test('exact instructions group runs while checkpoints and hold mode remain separate', () => {
  const a = validation('a');
  const b = validation('b', { checkpoint: 'policy/expert', mode: 'hold' });
  const groups = groupRolloutResults([a, b, validation('c', { instruction: 'Put the block in the box' })], []);
  assert.equal(groups.length, 2);
  const same = groups.find(group => group.instruction.endsWith('.'));
  assert.equal(same.runs.length, 2);
  assert.deepEqual(new Set(same.runs.map(run => run.checkpoint)), new Set(['policy/h0', 'policy/expert']));
  assert.deepEqual(new Set(same.runs.map(run => run.mode)), new Set(['real', 'hold']));
});

test('linked trials are not duplicated and missing validation records stay visible read-only', () => {
  const linked = trial('linked', undefined, { validation: { id: 'a' } });
  const orphan = trial('orphan', undefined, { validation: { id: 'missing' } });
  const legacy = trial('legacy');
  const grouped = validation('a', { attempts: [{ number: 1, rollout: linked, outcome: 'unjudged', notes: '' }] });
  const groups = groupRolloutResults([grouped], [linked, orphan, legacy]);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].runs.length, 3);
  assert.equal(groups[0].runs.find(run => run.id === 'orphan').kind, 'unlinked');
  assert.equal(groups[0].runs.find(run => run.id === 'legacy').kind, 'standalone');
  assert.equal(groups[0].runs.filter(run => run.key === 'standalone:linked').length, 0);
});

test('runs sort by actual time across recorded UTC offsets', () => {
  const groups = groupRolloutResults([
    validation('older', { created: '2026-11-26T09:00:00+09:00' }),
    validation('newer', { created: '2026-11-26T01:00:00Z' }),
  ], []);
  assert.deepEqual(groups[0].runs.map(run => run.id), ['newer', 'older']);
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
