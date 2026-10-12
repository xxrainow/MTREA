import { useState } from 'react';
import './TrainPage.css';

function TrainPage() {
  const [tab, setTab] = useState<'settings' | 'monitor'>('settings');
  const [dataset, setDataset] = useState('');
  const [seed, setSeed] = useState('0');
  const [steps, setSteps] = useState('30000');
  const [batchSize, setBatchSize] = useState('1');
  const [learningRate, setLearningRate] = useState('0.00005');
  const [runName, setRunName] = useState('h0_pickplace_seed0');

  return (
    <main className="train-page">
      <div className="train-heading">
        <h1>Train</h1>
        <p>Train a policy using collected data.</p>
      </div>

      <div className="train-tabs" role="tablist" aria-label="Training views">
        <button
          type="button"
          role="tab"
          id="train-settings-tab"
          aria-controls="train-settings"
          aria-selected={tab === 'settings'}
          onClick={() => setTab('settings')}
        >
          New training setup
        </button>
        <button
          type="button"
          role="tab"
          id="train-monitor-tab"
          aria-controls="train-monitor"
          aria-selected={tab === 'monitor'}
          onClick={() => setTab('monitor')}
        >
          Training monitor
        </button>
      </div>

      <section
        id="train-settings"
        role="tabpanel"
        aria-labelledby="train-settings-tab"
        hidden={tab !== 'settings'}
      >
        <div className="train-grid">
          <section className="train-panel">
            <h2>Training inputs</h2>

            <label className="train-field">
              Dataset
              <select
                value={dataset}
                onChange={(event) => setDataset(event.target.value)}
              >
                <option value="">Select a dataset</option>
                <option value="example">
                  pick_place_demo · T_sub · Example
                </option>
              </select>
            </label>

            <dl className="train-details">
              <div>
                <dt>Training method</dt>
                <dd>H0 · expert-only</dd>
              </div>
              <div>
                <dt>Policy</dt>
                <dd>π0.5</dd>
              </div>
              <div>
                <dt>Base model</dt>
                <dd>lerobot/pi05_base</dd>
              </div>
            </dl>

            <label className="train-field">
              Run name
              <input
                type="text"
                value={runName}
                onChange={(event) => setRunName(event.target.value)}
              />
            </label>
          </section>

          <section className="train-panel">
            <h2>Execution settings</h2>

            <div className="train-field-pair">
              <label className="train-field">
                Seed
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={seed}
                  onChange={(event) => setSeed(event.target.value)}
                />
              </label>

              <label className="train-field">
                Steps
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={steps}
                  onChange={(event) => setSteps(event.target.value)}
                />
              </label>
            </div>

            <div className="train-field-pair">
              <label className="train-field">
                Batch size
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={batchSize}
                  onChange={(event) => setBatchSize(event.target.value)}
                />
              </label>

              <label className="train-field">
                Learning rate
                <input
                  type="number"
                  min="0"
                  step="any"
                  value={learningRate}
                  onChange={(event) => setLearningRate(event.target.value)}
                />
              </label>
            </div>

            <dl className="train-details">
              <div>
                <dt>Execution location</dt>
                <dd>SSH · A100</dd>
              </div>
              <div>
                <dt>GPU server status</dt>
                <dd>Not checked</dd>
              </div>
            </dl>
          </section>
        </div>

        <section className="train-panel train-checks">
          <h2>Pre-run checks</h2>

          <dl className="train-details">
            <div>
              <dt>Dataset validation</dt>
              <dd>Not checked</dd>
            </div>
            <div>
              <dt>Normalization statistics</dt>
              <dd>Not checked</dd>
            </div>
            <div>
              <dt>Output directory conflict</dt>
              <dd>Not checked</dd>
            </div>
          </dl>

          <div className="train-actions">
            <button type="button" disabled>
              Check settings
            </button>
            <button type="button" className="train-button-primary" disabled>
              Start training
            </button>
          </div>
        </section>
      </section>

      <section
        id="train-monitor"
        role="tabpanel"
        aria-labelledby="train-monitor-tab"
        hidden={tab !== 'monitor'}
      >
        <div className="train-monitor-layout">
          <aside className="train-panel train-runs" aria-label="Training runs">
            <h2>Runs</h2>

            <div className="train-runs-list">
              <p className="train-empty">
                Training run list integration pending
              </p>
            </div>
          </aside>

          <div className="train-monitor-main">
            <section className="train-panel">
              <h2>Run status</h2>

              <dl className="train-details">
                <div>
                  <dt>Selected run</dt>
                  <dd>—</dd>
                </div>
                <div>
                  <dt>Step</dt>
                  <dd>—</dd>
                </div>
                <div>
                  <dt>Checkpoint</dt>
                  <dd>—</dd>
                </div>
              </dl>

              <button type="button" disabled>
                Stop training
              </button>
            </section>

            <div className="train-monitor-charts">
              <section className="train-panel">
                <h2>train/loss</h2>
                <div className="train-chart-placeholder">
                  Training metrics integration pending
                </div>
              </section>

              <section className="train-panel">
                <h2>train/learning_rate</h2>
                <div className="train-chart-placeholder">
                  Training metrics integration pending
                </div>
              </section>
            </div>

            <section className="train-panel">
              <h2>Saved checkpoints</h2>

              <div className="train-checkpoints-table-wrapper">
                <table className="train-checkpoints-table">
                  <thead>
                    <tr>
                      <th scope="col">Step</th>
                      <th scope="col">Location</th>
                      <th scope="col">Saved at</th>
                    </tr>
                  </thead>

                  <tbody>
                    <tr>
                      <td colSpan={3} className="train-checkpoints-empty">
                        Select a training run to view its saved checkpoints.
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>

            <section className="train-panel">
              <h2>Training logs</h2>
              <pre className="train-log">
                Select a training run to view its logs here.
              </pre>
            </section>
          </div>
        </div>
      </section>
    </main>
  );
}

export default TrainPage;
