import { hardwarePost, hardwareRequest } from './hardware';
import type { TrainJob } from '../training/model';

export type ServeRequest = {
  target: 'ssh' | 'local';
  mode: 'real' | 'hold';
  source_job_id: string | null;
  dataset_name: string | null;
};

export type PolicyCheckpoint = {
  job_id: string;
  step: number;
  label: string;
  checkpoint: string;
};

export type PolicyCheckpointsResponse = {
  checkpoints: PolicyCheckpoint[];
  errors: string[];
};

export type PolicySession = {
  job: TrainJob;
  state: string;

  status: {
    fake: boolean;
    checkpoint: string;
    client_connected: boolean;
    counts: Record<string, number>;
    last_error: string | null;
    last_refusal: string | null;
  } | null;

  session: Record<string, unknown> | null;
  unreachable: string | null;
  read_error: string | null;
  tunnel: TrainJob | null;
};

export function policySessionReady(session: PolicySession) {
  return ['ready', 'serving'].includes(session.state)
    && ['running', 'queued'].includes(session.job.status)
    && !session.unreachable && !session.read_error
    && (session.job.spec.executor === 'local' || session.tunnel?.status === 'running');
}

export function startPolicy(body: ServeRequest) {
  return hardwarePost<TrainJob>('/api/serve/start', body);
}

export function loadPolicyCheckpoints(signal?: AbortSignal) {
  return hardwareRequest<PolicyCheckpointsResponse>(
    '/api/serve/checkpoints',
    { signal },
  );
}

export function loadPolicySessions(signal?: AbortSignal) {
  return hardwareRequest<PolicySession[]>(
    '/api/serve/sessions',
    { signal },
  );
}

export function openPolicyTunnel(jobId: string) {
  return hardwarePost<TrainJob>(
    `/api/serve/sessions/${encodeURIComponent(jobId)}/tunnel`,
  );
}
