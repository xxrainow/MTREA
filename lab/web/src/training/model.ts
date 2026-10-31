// Pure helpers for the Train and Checkpoint pages. The form comes from
// GET /api/train/describe and everything about a run from its files on the
// executing machine (docs/TRAINING_JOBS.md); nothing here is simulated.

export type ParamType = 'string' | 'int' | 'float' | 'bool' | 'flag' | 'int_list';
export type ParamGroup = 'basic' | 'advanced' | 'fixed' | 'fake';
export type Param = {
  name: string; flag: string; type: ParamType; required: boolean; default: unknown;
  choices: string[] | null; min: number | null; label: string; help: string; group: ParamGroup;
};
export type Method = {
  name: string; label: string; description: string; max_batch_size_tested: number;
  oom_batch_size: number | null; seconds_per_step: Record<string, number>;
};
export type Describe = {
  params: Param[]; methods: Method[]; run_name_template: string;
  fixed: { lerobot_flags: Record<string, string>; scheduler: string };
};

export type Target = 'ssh' | 'local';
export type DatasetChoice = { backend: 'real' | 'fake'; name: string };
export type TrainingForm = {
  target: Target; fake: boolean; dataset: DatasetChoice | null;
  // Inputs as typed, by param name. Absent means "not set": train.sh's default applies.
  values: Record<string, string | boolean>;
  runName: string; runNameEdited: boolean;
};
export type TrainRequest = {
  target: Target; fake: boolean; dataset: DatasetChoice; run_name: string; params: Record<string, unknown>;
};

export type CheckItem = { id: string; status: 'ok' | 'warn' | 'fail'; message: string; link?: string };
export type CheckResult = {
  ok: boolean; checks: CheckItem[]; estimated_seconds: number | null; command: string | null;
  target: Target; host: string | null; run_dir: string; fake: boolean;
};

export type RunStatusFile = {
  status: 'running' | 'done' | 'failed' | 'stopped'; fake: boolean; method: string; seed: number;
  step: number; total_steps: number; loss: number | null; lr: number | null;
  seconds_per_step: number | null; eta_s: number | null; last_checkpoint_step: number | null;
  wandb_url: string | null; failure: 'oom' | 'error' | null; started_at: string; updated_at: string;
};
// status.json's word, or what the job's fate says when the file cannot.
export type RunState = RunStatusFile['status'] | 'starting' | 'ended_unknown' | 'queued' | 'unknown';
export type Metric = {
  step: number; loss: number; lr: number; grad_norm?: number; update_s?: number; data_s?: number;
  mem_gb?: number; time?: string;
};
export type CheckpointEntry = {
  step: number; path: string; complete: boolean; has_training_state: boolean; last: boolean; saved_at: string;
};
export type TrainJob = {
  id: string; status: string; pid: number | null; exit_code: number | null;
  submitted_at: string; started_at: string | null; finished_at: string | null;
  git_commit: string | null; unreachable: string | null;
  spec: { kind: string; cmd: string; executor: 'local' | 'ssh'; label: string; run_dir: string | null; host: string | null };
};
export type RunSummary = {
  job: TrainJob; status: RunStatusFile | null; state: RunState; checkpoint_count: number | null; unreachable: string | null;
};

// Set by lab from the target and dataset, never typed into the form.
export const LAB_PARAMS = new Set(['dataset', 'dataset_root', 'output', 'fake']);
const NAME = /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/;
export const METRICS_LIMIT = 2000;

export function initialForm(): TrainingForm {
  return { target: 'ssh', fake: false, dataset: null, values: {}, runName: '', runNameEdited: false };
}

export function formGroups(describe: Describe, fake: boolean) {
  const shown = describe.params.filter(p => !LAB_PARAMS.has(p.name));
  return {
    basic: shown.filter(p => p.group === 'basic'),
    advanced: shown.filter(p => p.group === 'advanced'),
    fixed: shown.filter(p => p.group === 'fixed'),
    fake: fake ? shown.filter(p => p.group === 'fake') : [],
  };
}

function pad(n: number) { return String(n).padStart(2, '0'); }

export function runNameFrom(template: string, method: string, dataset: string, seed: string, now: Date): string {
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}`;
  return template.replace('{method}', method || 'method').replace('{dataset_name}', dataset.split('/').at(-1) || 'dataset')
    .replace('{seed}', seed === '' ? 'X' : seed).replace('{yyyymmdd-HHMM}', stamp);
}

export function displayValue(param: Param, value: string | boolean | undefined): string | boolean {
  if (value !== undefined) return value;
  if (param.type === 'bool') return Boolean(param.default);
  if (param.default === null || param.default === undefined) return '';
  return Array.isArray(param.default) ? param.default.join(',') : String(param.default);
}

// Converts typed inputs to the request's params: only fields the user set.
export function toRequest(describe: Describe, form: TrainingForm): { request: TrainRequest | null; errors: string[] } {
  const errors: string[] = [];
  const params: Record<string, unknown> = {};
  const groups = formGroups(describe, form.fake);
  const editable = [...groups.basic, ...groups.advanced, ...groups.fake];
  for (const p of editable) {
    const raw = form.values[p.name];
    if (raw === undefined || raw === '') {
      if (p.required) errors.push(`${p.label} is required.`);
      continue;
    }
    if (p.type === 'bool') { params[p.name] = Boolean(raw); continue; }
    const text = String(raw).trim();
    if (p.type === 'int') {
      const n = Number(text);
      if (!/^-?\d+$/.test(text) || !Number.isSafeInteger(n)) { errors.push(`${p.label} must be a whole number.`); continue; }
      if (p.min !== null && n < p.min) { errors.push(`${p.label} must be at least ${p.min}.`); continue; }
      if (p.name === 'seed' && n < 0) { errors.push('Seed must be a non-negative integer.'); continue; }
      params[p.name] = n;
    } else if (p.type === 'float') {
      const n = Number(text);
      if (text === '' || !Number.isFinite(n)) { errors.push(`${p.label} must be a finite number.`); continue; }
      if (p.min !== null && n < p.min) { errors.push(`${p.label} must be at least ${p.min}.`); continue; }
      params[p.name] = n;
    } else if (p.type === 'int_list') {
      const items = text.split(',').map(s => s.trim()).filter(Boolean);
      if (items.some(s => !/^\d+$/.test(s))) { errors.push(`${p.label} must be comma-separated steps.`); continue; }
      params[p.name] = items.map(Number);
    } else {
      if (p.choices && !p.choices.includes(text)) { errors.push(`${p.label} must be one of ${p.choices.join(', ')}.`); continue; }
      params[p.name] = text;
    }
  }
  if (!form.dataset) errors.push('Select a dataset.');
  if (!NAME.test(form.runName)) errors.push('Run name may contain letters, digits, "_", "." and "-".');
  if (form.target === 'local' && !form.fake) errors.push('This machine runs fake training only.');
  if (errors.length || !form.dataset) return { request: null, errors };
  return { request: { target: form.target, fake: form.fake, dataset: form.dataset, run_name: form.runName, params }, errors };
}

// A check is valid for Start only while the inputs are exactly those checked.
export function requestKey(request: TrainRequest | null): string {
  return request ? JSON.stringify(request) : '';
}
export function canStart(check: CheckResult | null, checkedKey: string, currentKey: string): boolean {
  return Boolean(check?.ok) && checkedKey !== '' && checkedKey === currentKey;
}

export function isActiveState(state: RunState | null | undefined): boolean {
  return state === 'running' || state === 'starting' || state === 'queued';
}

export function stateLabel(state: RunState, failure?: RunStatusFile['failure']): string {
  switch (state) {
    case 'running': return 'Training';
    case 'starting': case 'queued': return 'Starting';
    case 'done': return 'Completed';
    case 'stopped': return 'Stopped';
    case 'failed': return failure === 'oom' ? 'Failed · Out of memory' : 'Failed';
    case 'ended_unknown': return 'Ended · status unknown';
    default: return 'Unknown';
  }
}

// Metrics arrive by byte offset, so a page never re-reads what it has.
export function appendMetrics(existing: Metric[], incoming: Metric[], limit = METRICS_LIMIT): Metric[] {
  if (!incoming.length) return existing;
  const last = existing.at(-1)?.step ?? -Infinity;
  const merged = [...existing, ...incoming.filter(m => m.step > last)];
  return merged.length > limit ? merged.slice(-limit) : merged;
}

export function progressPercent(step: number | undefined, total: number | undefined): number {
  if (!step || !total) return 0;
  return Math.min(100, Math.floor(step / total * 100));
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—';
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  if (h) return `${h} h ${m} min`;
  if (m) return `${m} min ${s % 60} s`;
  return `${s} s`;
}

export function sortCheckpoints(entries: CheckpointEntry[]): CheckpointEntry[] {
  return [...entries].sort((a, b) => b.step - a.step);
}

// "<job id>:<step>" links a checkpoint across pages; run identity is the job id.
export function checkpointId(jobId: string, step: number): string { return `${jobId}:${step}`; }
export function parseCheckpointId(id: string): { jobId: string; step: number } | null {
  const at = id.lastIndexOf(':');
  const step = Number(id.slice(at + 1));
  return at > 0 && Number.isSafeInteger(step) ? { jobId: id.slice(0, at), step } : null;
}
