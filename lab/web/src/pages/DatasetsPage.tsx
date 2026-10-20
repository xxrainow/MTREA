import { useEffect, useState } from 'react';
import { loadDataset, loadDatasets } from '../api/datasets';
import type { DatasetDetail, DatasetSummary } from '../api/datasets';
import DatasetEpisodeVideos from '../components/DatasetEpisodeVideos';
import './DatasetsPage.css';

function DatasetsPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selected, setSelected] = useState<DatasetSummary | null>(null);
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [episodeIndex, setEpisodeIndex] = useState(0);
  const [revision, setRevision] = useState(0);
  const [loadedRevision, setLoadedRevision] = useState(-1);
  const [error, setError] = useState('');

  useEffect(() => {
    const controller = new AbortController();
    loadDatasets(controller.signal).then((next) => {
      if (!controller.signal.aborted) {
        setDatasets(next);
        setError('');
        setLoadedRevision(revision);
      }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : 'Cannot load datasets.');
        setLoadedRevision(revision);
      }
    });
    return () => controller.abort();
  }, [revision]);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    loadDataset(selected, controller.signal).then((next) => {
      if (!controller.signal.aborted) {
        setDetail(next);
        setError('');
      }
    }).catch((reason: unknown) => {
      if (!controller.signal.aborted) {
        setError(reason instanceof Error ? reason.message : 'Cannot load dataset details.');
      }
    });
    return () => controller.abort();
  }, [selected]);

  function selectDataset(dataset: DatasetSummary) {
    setDetail(null);
    setError('');
    setEpisodeIndex(0);
    setSelected(dataset);
  }

  function backToList() {
    setSelected(null);
    setDetail(null);
    setError('');
    setRevision((value) => value + 1);
  }

  const loading = loadedRevision !== revision;

  if (selected === null) {
    return (
      <main className="datasets-page">
        <div className="datasets-heading">
          <div><h1>Datasets</h1><p>Browse locally collected demonstrations and metadata.</p></div>
          <button type="button" className="datasets-button" disabled={loading} onClick={() => setRevision((value) => value + 1)}>{loading ? 'Loading…' : 'Refresh'}</button>
        </div>
        {error && <p className="datasets-error" role="alert">{error}</p>}
        <section className="datasets-panel" aria-label="Dataset list" aria-busy={loading}>
          <div className="datasets-panel-header"><h2>Collected datasets</h2><span className="datasets-status">{datasets.length} {datasets.length === 1 ? 'dataset' : 'datasets'}</span></div>
          <div className="datasets-table-wrapper">
            <table className="datasets-table">
              <thead><tr><th scope="col">Dataset</th><th scope="col">Mode</th><th scope="col">Episodes</th><th scope="col">Frames</th><th scope="col">Metadata</th></tr></thead>
              <tbody>
                {datasets.map((dataset) => (
                  <tr key={dataset.id}>
                    <td><button type="button" className="datasets-name-button" onClick={() => selectDataset(dataset)} aria-label={`Open ${dataset.name} (${dataset.backend})`}>{dataset.name}</button></td>
                    <td><span className={`datasets-mode is-${dataset.backend}`}>{dataset.backend.toUpperCase()}</span></td>
                    <td>{dataset.episodes ?? '—'}</td><td>{dataset.total_frames ?? '—'}</td>
                    <td>{dataset.issues.length ? 'Needs attention' : 'Available'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!loading && !error && datasets.length === 0 && <p className="datasets-empty">No collected datasets yet. Finish a recording in Collect, then return here.</p>}
        </section>
        <p className="datasets-note datasets-catalog-note">Fake datasets contain metadata only. Metadata availability does not establish data quality or training compatibility.</p>
      </main>
    );
  }

  const dataset = detail?.dataset ?? selected;
  const episode = detail?.episodes[episodeIndex];

  return (
    <main className="datasets-page">
      <button type="button" className="datasets-back-button" onClick={backToList}>← Dataset list</button>
      <div className="datasets-heading">
        <div><h1>{dataset.name}</h1><p>Saved recording metadata and episode instructions</p></div>
        <span className={`datasets-mode is-${dataset.backend}`}>{dataset.backend.toUpperCase()}</span>
      </div>
      {error && <p className="datasets-error" role="alert">{error}</p>}
      {dataset.issues.map((issue) => <p className="datasets-error" role="alert" key={issue}>{issue}</p>)}
      <dl className="datasets-summary">
        <div><dt>Episodes</dt><dd>{dataset.episodes ?? '—'}</dd></div>
        <div><dt>Total frames</dt><dd>{dataset.total_frames ?? '—'}</dd></div>
        <div><dt>FPS</dt><dd>{dataset.fps ?? '—'}</dd></div>
        <div><dt>Robot</dt><dd>{dataset.robot_type ?? 'Not recorded'}</dd></div>
        <div><dt>Cameras</dt><dd>{dataset.cameras.join(', ') || 'Not recorded'}</dd></div>
      </dl>
      {!detail && !error && <p role="status">Loading episode metadata…</p>}
      {detail && <div className="datasets-browser">
        <section className="datasets-panel datasets-episodes" aria-label="Episode list">
          <div className="datasets-panel-header"><h2>Episodes</h2><span className="datasets-status">{detail.episodes.length}</span></div>
          <ul className="datasets-episode-list">
            {detail.episodes.map((item, index) => <li key={item.index}>
              <button type="button" className={`datasets-episode-button ${episodeIndex === index ? 'is-active' : ''}`} aria-pressed={episodeIndex === index} onClick={() => setEpisodeIndex(index)}>
                <span className="datasets-episode-index">{item.index}</span><span className="datasets-episode-name">Episode {item.index}</span>
              </button>
            </li>)}
          </ul>
          {detail.episodes.length === 0 && <p className="datasets-empty datasets-inset">No saved episode metadata.</p>}
        </section>
        <section className="datasets-panel datasets-viewer" aria-label="Dataset details">
          <div className="datasets-panel-header"><h2>{episode ? `Episode ${episode.index}` : 'Dataset details'}</h2></div>
          {episode && <DatasetEpisodeVideos key={`${dataset.id}/${episode.index}`} dataset={dataset} index={episode.index} />}
          <dl className="datasets-details">
            <div><dt>Instruction</dt><dd>{episode?.tasks.join(' · ') || 'Not recorded'}</dd></div>
            <div><dt>Dataset instructions</dt><dd>{detail.tasks.join(' · ') || 'Not recorded'}</dd></div>
            <div><dt>Storage location</dt><dd>{dataset.location}</dd></div>
          </dl>
          <p className="datasets-note">Episode numbers follow the saved dataset indices. Research roles are not inferred from web recordings.</p>
        </section>
      </div>}
    </main>
  );
}

export default DatasetsPage;
