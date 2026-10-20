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
  seed: number;
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
