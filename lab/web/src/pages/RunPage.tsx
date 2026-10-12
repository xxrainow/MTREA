import { useState } from 'react';
import './RunPage.css';

function RunPage() {
  const [checkpoint, setCheckpoint] = useState('');
  const [task, setTask] = useState('pick_place');
  const [instruction, setInstruction] = useState('');

  return (
    <main className="run-page">
      <div className="run-heading">
        <h1>Run & Evaluate</h1>
        <p>Run a trained policy and record rollout results.</p>
      </div>

      <div className="run-layout">
        <div className="run-main">
          <section className="run-panel">
            <div className="run-panel-header">
              <h2>Camera preview</h2>
              <span className="run-status">Integration pending</span>
            </div>

            <div className="run-cameras">
              {['wrist', 'scene'].map((camera) => (
                <figure className="run-camera" key={camera}>
                  <figcaption>{camera}</figcaption>
                  <div className="run-camera-placeholder">
                    Live video integration pending
                  </div>
                </figure>
              ))}
            </div>
          </section>

          <section className="run-panel">
            <h2>Run status</h2>

            <dl className="run-details">
              <div>
                <dt>Rollout ID</dt>
                <dd>—</dd>
              </div>
              <div>
                <dt>Status</dt>
                <dd>Not started</dd>
              </div>
              <div>
                <dt>Elapsed time</dt>
                <dd>—</dd>
              </div>
            </dl>

            <div className="run-actions">
              <button
                type="button"
                className="run-button run-button-primary"
                disabled
              >
                Start rollout
              </button>
              <button
                type="button"
                className="run-button"
                disabled
              >
                Stop execution
              </button>
            </div>
          </section>

          <section className="run-panel">
            <h2>Execution logs</h2>
            <pre className="run-log">
              Policy execution logs will appear here.
            </pre>
          </section>
        </div>

        <div className="run-controls">
          <section className="run-panel">
            <h2>Policy settings</h2>

            <label className="run-field">
              Checkpoint
              <select
                value={checkpoint}
                onChange={(event) =>
                  setCheckpoint(event.target.value)
                }
              >
                <option value="">Select a checkpoint</option>
                <option value="example">
                  h0_pickplace_seed0 · Example
                </option>
              </select>
            </label>

            <label className="run-field">
              Evaluation task
              <select
                value={task}
                onChange={(event) => setTask(event.target.value)}
              >
                <option value="pick_place">
                  pick_place · T_sub
                </option>
                <option value="stack">stack · held-out</option>
                <option value="push">push · held-out</option>
                <option value="open_drawer">
                  open_drawer · held-out
                </option>
                <option value="pour">pour · held-out</option>
              </select>
            </label>

            <label className="run-field">
              Task instruction
              <textarea
                rows={4}
                value={instruction}
                onChange={(event) =>
                  setInstruction(event.target.value)
                }
                placeholder="Task instruction to send to the policy"
              />
            </label>

            <dl className="run-details">
              <div>
                <dt>GPU server connection</dt>
                <dd>Not checked</dd>
              </div>
              <div>
                <dt>SSH tunnel</dt>
                <dd>Not checked</dd>
              </div>
              <div>
                <dt>Policy readiness</dt>
                <dd>Not checked</dd>
              </div>
              <div>
                <dt>Robot connection</dt>
                <dd>Not checked</dd>
              </div>
            </dl>

            <button
              type="button"
              className="run-button"
              disabled
            >
              Load policy
            </button>
          </section>

          <section className="run-panel">
            <h2>Rollout results</h2>

            <p className="run-description">
              Record whether the rollout succeeded after execution finishes.
            </p>

            <div className="run-actions">
              <button
                type="button"
                className="run-button"
                disabled
              >
                Success
              </button>
              <button
                type="button"
                className="run-button"
                disabled
              >
                Failure
              </button>
            </div>

            <p className="run-note">
              Result recording integration is pending. Results will be
              appended without deleting or overwriting existing records.
            </p>
          </section>
        </div>
      </div>

      <section className="run-panel run-history">
        <div className="run-panel-header">
          <h2>Evaluation history</h2>
          <span className="run-status">Integration pending</span>
        </div>

        <p className="run-description">
          Rollout records and absolute success rates for each task
          will be displayed here.
        </p>
      </section>
    </main>
  );
}

export default RunPage;
