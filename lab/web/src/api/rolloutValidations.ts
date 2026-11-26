import { hardwarePost, hardwareRequest } from './hardware';
import type { RolloutRequest, RolloutSnapshot } from './rollouts';

export type Outcome = 'success' | 'failure' | 'unjudged';
export type ValidationAttempt = { number: number; rollout: RolloutSnapshot; outcome: Outcome; notes: string };
export type RolloutValidation = {
  id: string;
  session: {
    configuration: RolloutRequest; target_count: number; checkpoint: string;
    server_address: string; created_at: string; instruction_key: string; git_commit: string | null;
    server_mode?: string; server_target?: string;
  };
  state: 'ready' | 'running' | 'awaiting_next' | 'awaiting_release' | 'completed' | 'ended' | 'blocked';
  attempts: ValidationAttempt[];
  started_count: number; released_count: number; judged_count: number;
  can_next: boolean; unresolved_requests: string[];
  start_errors: { error: string; request_id: string }[];
};
export type ValidationRequest = RolloutRequest & { target_count: number };
export const validationStateLabels: Record<RolloutValidation['state'], string> = {
  ready: 'Ready to start', running: 'Attempt in progress', awaiting_next: 'Ready for next attempt',
  awaiting_release: 'Release required', completed: 'Completed', ended: 'Ended early', blocked: 'Start unconfirmed',
};
const base = '/api/rollout-validations';
export const token = () => crypto.randomUUID().replaceAll('-', '');
export const loadValidations = (signal?: AbortSignal) => hardwareRequest<RolloutValidation[]>(base, { signal });
export const checkValidation = (body: ValidationRequest) =>
  hardwarePost<{ checkpoint: string; server_address: string }>(`${base}/check`, body);
export const startValidation = (body: ValidationRequest, requestId: string) =>
  hardwarePost<RolloutValidation>(`${base}/start`, { ...body, request_id: requestId });
export const nextValidationAttempt = (id: string, number: number, requestId: string) =>
  hardwarePost<RolloutValidation>(`${base}/${encodeURIComponent(id)}/next`, { number, request_id: requestId, scene_ready: true });
export const endValidation = (id: string) => hardwarePost<RolloutValidation>(`${base}/${encodeURIComponent(id)}/end`);
export const saveValidationOutcome = (id: string, jobId: string, outcome: Outcome, notes: string) =>
  hardwarePost<RolloutValidation>(`${base}/${encodeURIComponent(id)}/outcome`, { job_id: jobId, outcome, notes });
