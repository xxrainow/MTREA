import { hardwarePost, hardwareRequest } from './hardware';
import type { TrainJob } from '../training/model';

export type PolicyCheckpoint = { job_id: string; step: number; label: string; checkpoint: string };
export type PolicySession = {
  job: TrainJob; state: string; unreachable: string | null; client_address: string;
  tunnel: TrainJob | null;
  status: {
    fake: boolean; checkpoint: string; client_connected: boolean; failure: string | null;
    counts: Record<string, number>; last_error: string | null; last_refusal: string | null;
    last_latency_ms: Record<string, number> | null;
  } | null;
  session: Record<string, unknown> | null;
};
export type ServeRequest = {
  target: 'ssh' | 'local'; mode: 'real' | 'hold'; source_job_id: string | null; dataset_name: string | null;
};

export const loadPolicyCheckpoints = (signal?: AbortSignal) =>
  hardwareRequest<{ checkpoints: PolicyCheckpoint[]; errors: string[] }>('/api/serve/checkpoints', { signal });
export const loadPolicySessions = (signal?: AbortSignal) =>
  hardwareRequest<PolicySession[]>('/api/serve/sessions', { signal });
export const startPolicy = (body: ServeRequest) => hardwarePost<TrainJob>('/api/serve/start', body);
export const openPolicyTunnel = (jobId: string) =>
  hardwarePost<TrainJob>(`/api/serve/sessions/${encodeURIComponent(jobId)}/tunnel`);
