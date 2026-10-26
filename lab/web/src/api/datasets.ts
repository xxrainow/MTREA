import { hardwareRequest } from './hardware';

export type DatasetSummary = {
  id: string;
  name: string;
  backend: 'real' | 'fake';
  location: string;
  episodes: number | null;
  total_frames: number | null;
  fps: number | null;
  robot_type: string | null;
  cameras: string[];
  issues: string[];
};

export type DatasetDetail = {
  dataset: DatasetSummary;
  tasks: string[];
  episodes: { index: number; tasks: string[] }[];
};

export function loadDatasets(signal: AbortSignal) {
  return hardwareRequest<DatasetSummary[]>('/api/datasets', { signal });
}

export function loadDataset(dataset: DatasetSummary, signal: AbortSignal) {
  return hardwareRequest<DatasetDetail>(
    `/api/datasets/${dataset.backend}/${encodeURIComponent(dataset.name)}`, { signal },
  );
}

export type EpisodeVideo = { camera: string; start_s: number; end_s: number | null };

export function episodeVideoUrl(dataset: DatasetSummary, index: number, camera?: string) {
  const base = `/api/datasets/${dataset.backend}/${encodeURIComponent(dataset.name)}/episodes/${index}/videos`;
  return camera === undefined ? base : `${base}/${encodeURIComponent(camera)}`;
}

export function loadEpisodeVideos(dataset: DatasetSummary, index: number, signal: AbortSignal) {
  return hardwareRequest<EpisodeVideo[]>(episodeVideoUrl(dataset, index), { signal });
}


export type EpisodeJoints = {
  episode_index: number;
  joint_names: string[];
  frame_indices: number[];
  timestamps: number[];
  positions: (number | null)[][];
};

export function loadEpisodeJoints(
  dataset: DatasetSummary,
  index: number,
  signal: AbortSignal,
) {
  return hardwareRequest<EpisodeJoints>(
    `/api/datasets/${dataset.backend}/${encodeURIComponent(dataset.name)}/episodes/${index}/joints`,
    { signal },
  );
}

export type DatasetDeleteResult = {
  id: string;
  archive_id: string;
  archive_location: string;
};

export function deleteDataset(dataset: DatasetSummary) {
  return hardwareRequest<DatasetDeleteResult>(
    `/api/datasets/${dataset.backend}/${encodeURIComponent(dataset.name)}`,
    { method: 'DELETE' },
  );
}
