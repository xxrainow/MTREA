import { useEffect, useState } from 'react';
import { episodeVideoUrl, loadEpisodeVideos } from '../api/datasets';
import type { DatasetSummary, EpisodeVideo } from '../api/datasets';
import DatasetVideoPlayer from './DatasetVideoPlayer';

export default function DatasetEpisodeVideos({ dataset, index }: { dataset: DatasetSummary; index: number }) {
  const [clips, setClips] = useState<EpisodeVideo[] | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (dataset.backend === 'fake') return;
    const controller = new AbortController();
    loadEpisodeVideos(dataset, index, controller.signal).then((next) => {
      if (!controller.signal.aborted) setClips(next);
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Cannot load episode videos.');
    });
    return () => controller.abort();
  }, [dataset, index]);

  if (dataset.backend === 'fake') return <div className="datasets-video-placeholder"><div className="datasets-video-message"><strong>No video in Fake mode</strong><span>The simulated recording saves metadata, without camera frames or videos.</span></div></div>;
  if (error) return <p className="datasets-error" role="alert">{error}</p>;
  if (clips === null) return <p role="status">Loading episode videos…</p>;
  if (clips.length === 0) return <p className="datasets-empty">No recorded video cameras in this dataset.</p>;

  return <>
    <div className={`datasets-camera-grid ${clips.length > 1 ? 'show-all' : ''}`}>
      {clips.map((clip) => <DatasetVideoPlayer key={clip.camera} clip={clip} url={episodeVideoUrl(dataset, index, clip.camera)} />)}
    </div>
    <p className="datasets-note">Each camera plays independently. The timeline is relative to the selected episode.</p>
  </>;
}
