import { useState } from 'react';
import CameraPreviewPanel from '../components/CameraPreviewPanel';
import { useCameras } from '../hooks/useCameras';
import './CollectPage.css';

function CollectPage() {
  const [datasetName, setDatasetName] = useState('pick_place_demo');
  const [instruction, setInstruction] = useState(
    'Pick up the red block and place it in the blue cup.',
  );
  const [episodeCount, setEpisodeCount] = useState('10');
  const [episodeSeconds, setEpisodeSeconds] = useState('20');
  const [resetSeconds, setResetSeconds] = useState('5');

  const cameras = useCameras();
  const cameraStatus = cameras.data?.status;

  return (
    <main className="collect-page">
      <div className="collect-heading">
        <div>
          <h1>Collect</h1>
          <p>Record demonstrations through teleoperation.</p>
        </div>

        <button
          type="button"
          className="collect-button collect-button-primary"
          disabled
        >
          Start recording
        </button>
      </div>

      <div className="collect-layout">
        <div className="collect-main">
          <CameraPreviewPanel cameras={cameras} />
          <section className="collect-panel">
            <div className="collect-panel-header">
              <h2>Recording progress</h2>
              <span className="collect-status">Not started</span>
            </div>

            <dl className="collect-details">
              <div>
                <dt>Current episode</dt>
                <dd>— / {episodeCount || '—'}</dd>
              </div>
              <div>
                <dt>Time remaining</dt>
                <dd>—</dd>
              </div>
              <div>
                <dt>Status</dt>
                <dd>Recording integration pending</dd>
              </div>
            </dl>

            <div className="collect-actions">
              <button type="button" className="collect-button" disabled>
                Next episode
              </button>
              <button type="button" className="collect-button" disabled>
                Stop recording
              </button>
            </div>
          </section>

          <section className="collect-panel">
            <div className="collect-panel-header">
              <h2>Recording logs</h2>
              <span className="collect-status">Integration pending</span>
            </div>

            <pre className="collect-log">
              Logs will appear here when recording starts.
            </pre>
          </section>
        </div>

        <section className="collect-panel">
          <div className="collect-panel-header">
            <h2>Collection settings</h2>
          </div>

          <dl className="collect-details">
            <div>
              <dt>Task</dt>
              <dd>pick_place · T_sub</dd>
            </div>
            <div>
              <dt>Collection plan</dt>
              <dd>Plan file integration pending</dd>
            </div>
            <div>
              <dt>Current object</dt>
              <dd>Plan file integration pending</dd>
            </div>
          </dl>

          <label className="collect-field">
            Dataset name
            <input
              type="text"
              value={datasetName}
              onChange={(event) => setDatasetName(event.target.value)}
            />
          </label>

          <label className="collect-field">
            Task instruction
            <textarea
              rows={4}
              value={instruction}
              onChange={(event) => setInstruction(event.target.value)}
            />
          </label>

          <label className="collect-field">
            Number of episodes
            <input
              type="number"
              min="1"
              step="1"
              value={episodeCount}
              onChange={(event) => setEpisodeCount(event.target.value)}
            />
          </label>

          <div className="collect-field-pair">
            <label className="collect-field">
              Recording duration (seconds)
              <input
                type="number"
                min="1"
                step="1"
                value={episodeSeconds}
                onChange={(event) => setEpisodeSeconds(event.target.value)}
              />
            </label>

            <label className="collect-field">
              Reset duration (seconds)
              <input
                type="number"
                min="0"
                step="1"
                value={resetSeconds}
                onChange={(event) => setResetSeconds(event.target.value)}
              />
            </label>
          </div>

          <dl className="collect-details">
            <div>
              <dt>Robot</dt>
              <dd>Not checked</dd>
            </div>
            <div>
              <dt>Cameras</dt>
              <dd>
                {cameras.unavailable
                  ? 'Status unavailable'
                  : cameraStatus?.error
                    ? 'Preview failed'
                    : cameraStatus?.running
                      ? cameraStatus.cameras.join(', ')
                      : 'Preview stopped'}
              </dd>
            </div>
          </dl>
        </section>
      </div>
    </main>
  );
}

export default CollectPage;
