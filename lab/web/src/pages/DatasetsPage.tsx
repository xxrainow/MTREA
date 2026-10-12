import { useState } from 'react';
import './DatasetsPage.css';

type DatasetPreview = {
  id: string;
  name: string;
  task: string;
  role: string;
  episodes: number;
};

type CameraView = 'wrist' | 'scene' | 'all';

const exampleDatasets: DatasetPreview[] = [
  {
    id: 'demo',
    name: 'pick_place_demo',
    task: 'pick_place',
    role: 'T_sub',
    episodes: 10,
  },
  {
    id: 'test',
    name: 'pick_place_cli_test',
    task: 'pick_place',
    role: 'T_sub',
    episodes: 1,
  },
];

function DatasetsPage() {
  const [selectedDataset, setSelectedDataset] =
    useState<DatasetPreview | null>(null);
  const [episodeIndex, setEpisodeIndex] = useState(0);
  const [cameraView, setCameraView] = useState<CameraView>('scene');

  function selectDataset(dataset: DatasetPreview) {
    setSelectedDataset(dataset);
    setEpisodeIndex(0);
    setCameraView('scene');
  }

  if (selectedDataset === null) {
    return (
      <main className="datasets-page">
        <div className="datasets-heading">
          <div>
            <h1>Datasets</h1>
            <p>
              Explore collected demonstrations and check dataset status.
            </p>
          </div>

          <span className="datasets-status">Example data</span>
        </div>

        <section className="datasets-panel">
          <div className="datasets-panel-header">
            <h2>Dataset list</h2>
            <button
              type="button"
              className="datasets-button"
              disabled
            >
              Refresh
            </button>
          </div>

          <div className="datasets-table-wrapper">
            <table className="datasets-table">
              <thead>
                <tr>
                  <th scope="col">Dataset</th>
                  <th scope="col">Task</th>
                  <th scope="col">Role</th>
                  <th scope="col">Episodes</th>
                  <th scope="col">Validation</th>
                </tr>
              </thead>

              <tbody>
                {exampleDatasets.map((dataset) => (
                  <tr key={dataset.id}>
                    <td>
                      <button
                        type="button"
                        className="datasets-name-button"
                        onClick={() => selectDataset(dataset)}
                      >
                        {dataset.name}
                      </button>
                    </td>
                    <td>{dataset.task}</td>
                    <td>{dataset.role}</td>
                    <td>{dataset.episodes}</td>
                    <td>Not checked</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <p className="datasets-empty">
          Select a dataset to browse its episodes.
        </p>
      </main>
    );
  }

  const visibleCameras =
    cameraView === 'all' ? ['wrist', 'scene'] : [cameraView];

  return (
    <main className="datasets-page">
      <button
        type="button"
        className="datasets-back-button"
        onClick={() => setSelectedDataset(null)}
      >
        ← Dataset list
      </button>

      <div className="datasets-heading">
        <div className="datasets-title">
          <h1>{selectedDataset.name}</h1>
          <p>Recorded demonstrations and episode browser</p>
        </div>

        <div className="datasets-heading-actions">
          <label className="datasets-field datasets-selector">
            Dataset
            <select
              value={selectedDataset.id}
              onChange={(event) => {
                const dataset = exampleDatasets.find(
                  (item) => item.id === event.target.value,
                );

                if (dataset) {
                  selectDataset(dataset);
                }
              }}
            >
              {exampleDatasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className="datasets-button"
            disabled
          >
            Upload to Hub
          </button>
        </div>
      </div>

      <dl className="datasets-summary">
        <div>
          <dt>Episodes</dt>
          <dd>{selectedDataset.episodes}</dd>
        </div>
        <div>
          <dt>Total frames</dt>
          <dd>—</dd>
        </div>
        <div>
          <dt>FPS</dt>
          <dd>—</dd>
        </div>
        <div>
          <dt>Robot</dt>
          <dd>SO-101 · example</dd>
        </div>
        <div>
          <dt>Cameras</dt>
          <dd>wrist, scene · example</dd>
        </div>
        <div className="datasets-summary-validation">
          <dt>Validation</dt>
          <dd>Not checked</dd>
        </div>
      </dl>

      <div className="datasets-browser">
        <section
          className="datasets-panel datasets-episodes"
          aria-label="Episode list"
        >
          <div className="datasets-panel-header">
            <h2>Episodes</h2>
            <span className="datasets-status">
              {selectedDataset.episodes}
            </span>
          </div>

          <ul className="datasets-episode-list">
            {Array.from(
              { length: selectedDataset.episodes },
              (_, index) => (
                <li key={index}>
                  <button
                    type="button"
                    className={`datasets-episode-button ${
                      episodeIndex === index ? 'is-active' : ''
                    }`}
                    aria-pressed={episodeIndex === index}
                    onClick={() => setEpisodeIndex(index)}
                  >
                    <span className="datasets-episode-index">
                      {index}
                    </span>

                    <span className="datasets-episode-name">
                      Episode {index}
                    </span>

                    <span className="datasets-episode-duration">
                      —
                    </span>
                  </button>
                </li>
              ),
            )}
          </ul>
        </section>

        <section
          className="datasets-panel datasets-viewer"
          aria-label={`Episode ${episodeIndex} viewer`}
        >
          <div className="datasets-viewer-toolbar">
            <div
              className="datasets-camera-options"
              role="group"
              aria-label="Camera view"
            >
              {(
                [
                  ['wrist', 'wrist'],
                  ['scene', 'scene'],
                  ['all', 'All cameras'],
                ] as const
              ).map(([value, label]) => (
                <button
                  type="button"
                  key={value}
                  className={`datasets-button ${
                    cameraView === value ? 'is-active' : ''
                  }`}
                  aria-pressed={cameraView === value}
                  onClick={() => setCameraView(value)}
                >
                  {label}
                </button>
              ))}
            </div>

            <div
              className="datasets-playback-options"
              role="group"
              aria-label="Playback controls"
            >
              <button
                type="button"
                className="datasets-button"
                disabled
              >
                0.5x
              </button>
              <button
                type="button"
                className="datasets-button"
                disabled
              >
                1x
              </button>
              <button
                type="button"
                className="datasets-button"
                disabled
              >
                2x
              </button>
              <button
                type="button"
                className="datasets-button"
                disabled
              >
                Fullscreen
              </button>
            </div>
          </div>

          <div
            className={`datasets-camera-grid ${
              cameraView === 'all' ? 'show-all' : ''
            }`}
          >
            {visibleCameras.map((camera) => (
              <div
                key={`${selectedDataset.id}-${episodeIndex}-${camera}`}
                className="datasets-video-placeholder"
                aria-label={`${camera} camera, episode ${episodeIndex}`}
              >
                <span className="datasets-camera-label">
                  {camera}
                </span>

                <div className="datasets-video-message">
                  <strong>Episode {episodeIndex}</strong>
                  <span>Recorded video integration pending</span>
                </div>
              </div>
            ))}
          </div>

          <div className="datasets-player-controls">
            <button
              type="button"
              className="datasets-button"
              disabled
            >
              Play
            </button>

            <button
              type="button"
              className="datasets-button"
              aria-label="Previous frame"
              disabled
            >
              Previous frame
            </button>

            <button
              type="button"
              className="datasets-button"
              aria-label="Next frame"
              disabled
            >
              Next frame
            </button>

            <label className="datasets-timeline">
              <span className="datasets-sr-only">
                Playback position
              </span>
              <input
                type="range"
                min="0"
                max="100"
                value="0"
                disabled
              />
            </label>

            <span className="datasets-player-time">— / —</span>
          </div>

          <section className="datasets-joint-panel">
            <div className="datasets-panel-header">
              <h3>Joint positions</h3>
              <span className="datasets-description">Frame —</span>
            </div>

            <div className="datasets-chart-placeholder">
              Joint data visualization integration pending
            </div>
          </section>

          <dl className="datasets-details">
            <div>
              <dt>Task</dt>
              <dd>{selectedDataset.task}</dd>
            </div>
            <div>
              <dt>Role</dt>
              <dd>{selectedDataset.role}</dd>
            </div>
            <div>
              <dt>Storage location</dt>
              <dd>Integration pending</dd>
            </div>
          </dl>

          <p className="datasets-note">
            Example data only. Video playback, joint data, file access,
            and uploads are not yet integrated.
          </p>
        </section>
      </div>
    </main>
  );
}

export default DatasetsPage;
