import { hardwarePost, hardwareRequest } from './hardware';
import type {
  CheckpointEntry, CheckResult, Describe, Metric, RunStatusFile, RunState, RunSummary, TrainJob, TrainRequest,
} from '../training/model';

export type Targets = {
  local: boolean;
  ssh: { configured: boolean; reachable: boolean; host: string | null; message: string };
};
export type RunStatusResponse = { job: TrainJob; status: RunStatusFile | null; state: RunState };
export type CheckpointsResponse = {
  run_dir: string; host: string | null; checkpoints: { checkpoints: CheckpointEntry[]; updated_at: string } | null;
};

const run = (jobId: string) => `/api/train/runs/${encodeURIComponent(jobId)}`;

export function loadDescribe(signal?: AbortSignal) {
  return hardwareRequest<Describe>('/api/train/describe', { signal });
}

export function loadTargets(signal?: AbortSignal) {
  return hardwareRequest<Targets>('/api/train/targets', { signal });
}

export function checkTraining(request: TrainRequest) {
  return hardwarePost<CheckResult>('/api/train/check', request);
}

export class StartRefused extends Error {
  check: CheckResult | null;
  constructor(message: string, check: CheckResult | null) {
    super(message);
    this.check = check;
  }
}

// 409 carries either the failed checks (re-run on the server) or a busy GPU.
export async function startTraining(request: TrainRequest): Promise<TrainJob> {
  const response = await fetch('/api/train/start', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request),
  });
  const body = await response.json().catch(() => null);
  if (response.ok) return body as TrainJob;
  const detail = body?.detail;
  if (detail && typeof detail === 'object' && Array.isArray(detail.checks)) {
    throw new StartRefused('Checks failed on the target. Review them and check again.', detail as CheckResult);
  }
  throw new StartRefused(typeof detail === 'string' ? detail : `Request failed: HTTP ${response.status}`, null);
}

export function loadRuns(signal?: AbortSignal) {
  return hardwareRequest<RunSummary[]>('/api/train/runs', { signal });
}

export function loadRunStatus(jobId: string, signal?: AbortSignal) {
  return hardwareRequest<RunStatusResponse>(`${run(jobId)}/status`, { signal });
}

export function loadMetrics(jobId: string, offset: number, signal?: AbortSignal) {
  return hardwareRequest<{ metrics: Metric[]; offset: number }>(`${run(jobId)}/metrics?offset=${offset}`, { signal });
}

export function loadCheckpoints(jobId: string, signal?: AbortSignal) {
  return hardwareRequest<CheckpointsResponse>(`${run(jobId)}/checkpoints`, { signal });
}

export function loadRunConfig(jobId: string, signal?: AbortSignal) {
  return hardwareRequest<{ config: Record<string, unknown> | null }>(`${run(jobId)}/config`, { signal });
}

export function loadJobLog(jobId: string, signal?: AbortSignal) {
  return hardwareRequest<{ content: string }>(`/api/jobs/${encodeURIComponent(jobId)}/logs?tail_lines=200`, { signal });
}

export function stopJob(jobId: string) {
  return hardwarePost<TrainJob>(`/api/jobs/${encodeURIComponent(jobId)}/stop`);
}
