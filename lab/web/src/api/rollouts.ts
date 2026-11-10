import { hardwarePost, hardwareRequest } from './hardware';
import type { TrainJob } from '../training/model';

export type RolloutRequest = {
  server_job_id: string;
  instruction: string;
  max_relative_target: number;
  time_limit_s: number;
};
export type RolloutSnapshot = {
  job: TrainJob;
  state: string;
  owner_alive: boolean;
  read_error: string | null;
  status: {
    state: string;
    elapsed_s: number;
    workers_stopped?: boolean;
    arm_released?: boolean;
    hold_command_sent?: boolean;
    reason?: string | null;
    error?: string | null;
    release_error?: string | null;
    actions_executed?: number;
    chunks_received?: number;
    queue_size?: number;
    last_latency_ms?: number | null;
    stalls?: number;
    stalling?: boolean;
    preview_active?: boolean;
    preview_error?: string | null;
    frames?: Record<string, { available: boolean; captured_at: number | null }>;
  } | null;
  session: { checkpoint: string; task: string; server_address: string } | null;
};
export type CommandResult = {
  id: string;
  state: 'queued' | 'accepted' | 'completed' | 'rejected' | 'failed';
  error?: string | null;
};
export type QueuedCommand = { job_id: string; command_id: string; state: 'queued' };

export const loadRollouts = (signal?: AbortSignal) =>
  hardwareRequest<RolloutSnapshot[]>('/api/rollouts', { signal });
export const checkRollout = (body: RolloutRequest) =>
  hardwarePost<{ checkpoint: string; server_address: string }>('/api/rollouts/check', body);
export const startRollout = (body: RolloutRequest) =>
  hardwarePost<RolloutSnapshot>('/api/rollouts/start', body);
export const sendRolloutCommand = (id: string, action: 'stop' | 'release') =>
  hardwarePost<QueuedCommand>(`/api/rollouts/${encodeURIComponent(id)}/${action}`);
export const loadRolloutCommand = (jobId: string, commandId: string, signal?: AbortSignal) =>
  hardwareRequest<CommandResult>(
    `/api/rollouts/${encodeURIComponent(jobId)}/commands/${encodeURIComponent(commandId)}`, { signal });
export const loadRolloutLog = (id: string, signal?: AbortSignal) =>
  hardwareRequest<{ content: string }>(`/api/rollouts/${encodeURIComponent(id)}/logs`, { signal });

export function rolloutOwnsDevices(item: RolloutSnapshot) {
  return item.owner_alive || ['running', 'queued'].includes(item.job.status);
}
