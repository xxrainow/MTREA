import { hardwarePost, hardwareRequest } from './hardware';
import type { DatasetReference } from './datasets';

export type RemoteStatus = {
  configured: boolean;
  reachable: boolean;
  message: string;
  host: string | null;
};

export type RemoteState = 'absent' | 'incoming' | 'synced' | 'different' | 'unknown';

export type DatasetRemote = {
  sendable: boolean;
  reason: string | null;
  state: RemoteState;
  message: string;
  // Where the dataset lives on the GPU host; --dataset-root for training once synced.
  remote_path: string | null;
  local_fingerprint: string | null;
  remote_fingerprint: string | null;
  synced_at: string | null;
};

export type RemoteMarker =
  | 'synced'
  | 'different'
  | 'not_synced'
  | 'not_sent_fake'
  | 'unknown';

export type SyncJob = {
  id: string;
  status: string;
  spec: { kind: string; label: string };
};

export type SyncProgress = { phase: 'hash' | 'transfer' | 'verify'; percent: number };

function datasetPath(dataset: DatasetReference) {
  return `/api/datasets/${dataset.backend}/${encodeURIComponent(dataset.name)}`;
}

export function loadRemoteStatus(signal?: AbortSignal) {
  return hardwareRequest<RemoteStatus>('/api/remote/status', { signal });
}

export function loadRemoteMarkers(signal?: AbortSignal) {
  return hardwareRequest<Record<string, RemoteMarker>>('/api/remote/datasets', { signal });
}

export function loadDatasetRemote(dataset: DatasetReference, signal?: AbortSignal) {
  return hardwareRequest<DatasetRemote>(`${datasetPath(dataset)}/remote`, { signal });
}

export function syncDataset(dataset: DatasetReference) {
  return hardwarePost<SyncJob>(`${datasetPath(dataset)}/sync`);
}

export function syncJobLabel(dataset: DatasetReference) {
  return `sync ${dataset.name}`;
}

export async function findActiveSyncJob(dataset: DatasetReference, signal?: AbortSignal) {
  const jobs = await hardwareRequest<SyncJob[]>('/api/jobs', { signal });
  return (
    jobs.find(
      (job) =>
        job.spec.kind === 'sync' &&
        job.spec.label === syncJobLabel(dataset) &&
        job.status === 'running',
    ) ?? null
  );
}

export function loadJob(jobId: string, signal?: AbortSignal) {
  return hardwareRequest<SyncJob>(`/api/jobs/${encodeURIComponent(jobId)}`, { signal });
}

// The last PROGRESS line of lab.core.dataset_sync in the job log.
export async function loadSyncProgress(jobId: string, signal?: AbortSignal) {
  const log = await hardwareRequest<{ content: string }>(
    `/api/jobs/${encodeURIComponent(jobId)}/logs?tail_lines=200`,
    { signal },
  );
  let progress: SyncProgress | null = null;
  for (const line of log.content.split('\n')) {
    const match = /^PROGRESS phase=(hash|transfer|verify) percent=(\d+)/.exec(line);
    if (match) {
      progress = { phase: match[1] as SyncProgress['phase'], percent: Number(match[2]) };
    }
  }
  return progress;
}
