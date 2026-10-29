import { hardwarePost, hardwareRequest } from './hardware';

export type CollectionInput = {
  dataset: string;
  instruction: string;
  num_episodes: number;
  episode_time_s: number;
  reset_time_s: number;
};

export type CollectionSessionInput = Omit<
  CollectionInput,
  'dataset'
>;

export type CollectionStartInput = {
  dataset: string;
  session_index: number;
  sessions: CollectionSessionInput[];
};

export type CollectionPhase =
  | 'idle'
  | 'starting'
  | 'recording'
  | 'stopping'
  | 'verifying'
  | 'finished'
  | 'failed';

export type CollectionStatus = {
  phase: CollectionPhase;
  session_index: number | null;
  error: string | null;
  stop_requested: boolean;
  backend: 'real' | 'fake';
  dataset: string | null;
  instruction: string | null;
  current_episode: number | null;
  total_episodes: number;
  saved_episodes: number;
  episode_phase: string | null;
  phase_id: string | null;
  phase_elapsed_s: number;
  phase_duration_s: number | null;
  session_elapsed_s: number;
  camera_names: string[];
  recording_id: string | null;
  preview_error: string | null;
};

export function validateCollectionInput(
  input: CollectionInput,
): Promise<CollectionInput> {
  return hardwarePost<CollectionInput>(
    '/api/collection/validate',
    input,
  );
}

export function startCollection(
  input: CollectionStartInput,
): Promise<CollectionStatus> {
  return hardwarePost<CollectionStatus>(
    '/api/collection/start',
    input,
  );
}

export function stopCollection(): Promise<CollectionStatus> {
  return hardwarePost<CollectionStatus>(
    '/api/collection/stop',
  );
}

export function loadCollectionStatus(
  signal: AbortSignal,
): Promise<CollectionStatus> {
  return hardwareRequest<CollectionStatus>(
    '/api/collection/status',
    { signal },
  );
}


export type CollectionCommandInput = {
  action: 'start_episode' | 'finish_episode' | 'retry_episode';
  recording_id: string;
  expected_phase_id: string;
  expected_phase: 'preparing' | 'recording' | 'resetting';
  expected_episode: number;
};

export type CollectionCommandResult = {
  command_id: string;
  status: 'queued';
};

export function sendCollectionCommand(
  input: CollectionCommandInput,
): Promise<CollectionCommandResult> {
  return hardwarePost<CollectionCommandResult>(
    '/api/collection/command',
    input,
  );
}
