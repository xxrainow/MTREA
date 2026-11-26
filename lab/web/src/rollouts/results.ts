import type { RolloutSnapshot } from '../api/rollouts';
import type { RolloutValidation, ValidationAttempt } from '../api/rolloutValidations';

export type RolloutResultTarget = { validationId?: string; jobId?: string };
export type ResultRun = {
  key: string; id: string; kind: 'validation' | 'standalone' | 'unlinked'; instruction: string;
  createdAt: string; checkpoint: string; mode: string; state: string;
  targetCount: number; attempts: ValidationAttempt[]; validation?: RolloutValidation;
};
export type ResultSelection = { runKey?: string; jobId?: string };

export function resultTargetSelection(target: RolloutResultTarget): ResultSelection {
  if (!target.validationId && !target.jobId) return {};
  return { runKey: target.validationId ? `validation:${target.validationId}` : `standalone:${target.jobId}`, jobId: target.jobId };
}

export function listRolloutResults(validations: RolloutValidation[], trials: RolloutSnapshot[]): ResultRun[] {
  const linked = new Set(validations.flatMap(item => item.attempts.map(attempt => attempt.rollout.job.id)));
  const runs: ResultRun[] = validations.map(item => ({
    key: `validation:${item.id}`, id: item.id, kind: 'validation',
    instruction: item.session.configuration.instruction, createdAt: item.session.created_at,
    checkpoint: item.session.checkpoint, mode: item.session.server_mode ?? 'unknown', state: item.state,
    targetCount: item.session.target_count, attempts: item.attempts, validation: item,
  }));
  for (const trial of trials) {
    if (linked.has(trial.job.id)) continue;
    const unlinked = Boolean(trial.job.spec.metadata?.validation);
    runs.push({ key: `standalone:${trial.job.id}`, id: trial.job.id, kind: unlinked ? 'unlinked' : 'standalone',
      instruction: trial.session?.task ?? String(trial.job.spec.metadata?.instruction ?? trial.job.spec.label ?? 'Instruction not recorded'),
      createdAt: trial.job.submitted_at,
      checkpoint: trial.session?.checkpoint ?? String(trial.job.spec.metadata?.checkpoint ?? 'Not recorded'),
      mode: String(trial.job.spec.metadata?.mode ?? 'unknown'), state: trial.state, targetCount: 1,
      attempts: [{ number: 1, rollout: trial, outcome: 'unjudged', notes: '' }],
    });
  }
  runs.sort((a, b) => (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0)
    || a.key.localeCompare(b.key));
  return runs;
}

export function outcomeCounts(attempts: ValidationAttempt[]) {
  return attempts.reduce((counts, attempt) => {
    counts[attempt.outcome] += 1;
    return counts;
  }, { success: 0, failure: 0, unjudged: 0 });
}

export function formatResultDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export function formatElapsed(value: unknown) {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)} s` : '—';
}
