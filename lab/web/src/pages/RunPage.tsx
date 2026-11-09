import { useEffect, useState } from 'react';
import { errorMessage } from '../api/hardware';
import {
  loadPolicyCheckpoints,
  loadPolicySessions,
  openPolicyTunnel,
  startPolicy,
} from '../api/serving';
import type { PolicyCheckpoint, PolicySession } from '../api/serving';
import { loadJobLog, stopJob } from '../api/training';

import './RunPage.css';

function RunPage() {
  const [checkpoint, setCheckpoint] = useState('');
  const [task, setTask] = useState('pick_place');
  const [instruction, setInstruction] = useState('');
  const [sessions, setSessions] = useState<PolicySession[] | null>(null);
  const [sessionError, setSessionError] = useState('');

  const [checkpoints, setCheckpoints] = useState<PolicyCheckpoint[] | null>(
    null,
  );
  const [checkpointError, setCheckpointError] = useState('');

  const [startingPolicy, setStartingPolicy] = useState(false);
  const [policyError, setPolicyError] = useState('');
  const [stoppingPolicyId, setStoppingPolicyId] = useState<string | null>(null);

  const [logJobId, setLogJobId] = useState<string | null>(null);
  const [logContent, setLogContent] = useState('');
  const [logError, setLogError] = useState('');
  const [logLoading, setLogLoading] = useState(false);

  const [openingTunnelId, setOpeningTunnelId] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      try {
        const next = await loadPolicySessions(controller.signal);

        if (controller.signal.aborted) return;

        setSessions(next);
        setSessionError('');
      } catch (reason) {
        if (!controller.signal.aborted) {
          setSessionError(errorMessage(reason));
        }
      }

      if (!controller.signal.aborted) {
        timer = setTimeout(poll, 2000);
      }
    }

    void poll();

    return () => {
      controller.abort();

      if (timer !== undefined) {
        clearTimeout(timer);
      }
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();

    loadPolicyCheckpoints(controller.signal)
      .then((next) => {
        setCheckpoints(next.checkpoints);
        setCheckpointError(next.errors.join('\n'));
      })
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setCheckpointError(errorMessage(reason));
        }
      });

    return () => controller.abort();
  }, []);

  const policyRunning =
    sessions?.some(
      (session) =>
        session.job.status === 'running' || session.job.status === 'queued',
    ) ?? false;

  const canStartPolicy =
    Boolean(checkpoints?.some((item) => item.job_id === checkpoint)) &&
    sessions !== null &&
    !sessionError &&
    !policyRunning &&
    !startingPolicy;

  async function handleStartPolicy() {
    if (!canStartPolicy) return;

    setStartingPolicy(true);
    setPolicyError('');

    try {
      await startPolicy({
        target: 'ssh',
        mode: 'real',
        source_job_id: checkpoint,
        dataset_name: null,
      });
    } catch (reason) {
      setPolicyError(errorMessage(reason));
    } finally {
      setStartingPolicy(false);
    }
  }

  async function handleStopPolicy(jobId: string) {
    if (stoppingPolicyId !== null) return;

    setStoppingPolicyId(jobId);
    setPolicyError('');

    try {
      await stopJob(jobId);
    } catch (reason) {
      setPolicyError(errorMessage(reason));
    } finally {
      setStoppingPolicyId(null);
    }
  }

  async function handleViewLogs(jobId: string) {
    if (logLoading) return;

    if (logJobId === jobId) {
      setLogJobId(null);
      return;
    }

    setLogJobId(jobId);
    setLogLoading(true);
    setLogContent('');
    setLogError('');

    try {
      const result = await loadJobLog(jobId);
      setLogContent(result.content);
    } catch (reason) {
      setLogError(errorMessage(reason));
    } finally {
      setLogLoading(false);
    }
  }

  async function handleOpenTunnel(jobId: string) {
    if (openingTunnelId !== null) return;

    setOpeningTunnelId(jobId);
    setPolicyError('');

    try {
      await openPolicyTunnel(jobId);
    } catch (reason) {
      setPolicyError(errorMessage(reason));
    } finally {
      setOpeningTunnelId(null);
    }
  }

  return (
    <main className="run-page">
      <div className="run-heading">
        <h1>Run & Evaluate</h1>
        <p>Run a trained policy and record rollout results.</p>
      </div>

      <div className="run-layout">
        <div className="run-main">
          <section className="run-panel">
            <h2>Policy server sessions</h2>

            {sessionError ? (
              <p role="alert">{sessionError}</p>
            ) : sessions === null ? (
              <p>Loading sessions…</p>
            ) : sessions.length === 0 ? (
              <p>No policy server sessions yet.</p>
            ) : (
              <ul className="run-sessions">
                {sessions.map((session) => (
                  <li key={session.job.id} className="run-session">
                    <div>
                      {session.job.spec.label} · {session.state}
                    </div>

                    <p className="run-description">
                      Checkpoint:{' '}
                      {session.status?.checkpoint ??
                        String(
                          session.job.spec.metadata?.checkpoint ?? 'Unknown',
                        )}
                      {!session.status?.checkpoint && (
                        <span> · requested; loading not confirmed</span>
                      )}
                    </p>

                    {session.unreachable && (
                      <p role="alert">
                        Cannot reach the GPU host: {session.unreachable} The
                        current server state could not be verified.
                      </p>
                    )}

                    {session.read_error && (
                      <p role="alert">
                        Cannot read session files: {session.read_error}
                      </p>
                    )}

                    {session.status?.last_error && (
                      <p role="alert">
                        Prediction error: {session.status.last_error}
                      </p>
                    )}

                    {session.status?.last_refusal && (
                      <p role="status">
                        Last refused connection: {session.status.last_refusal}
                      </p>
                    )}
                    <p className="run-description">
                      SSH tunnel:{' '}
                      {session.tunnel ? session.tunnel.status : 'Not opened'}
                    </p>

                    {session.tunnel?.status === 'running' && (
                      <p className="run-description">
                        SSH process is running. Inference link is not verified.
                      </p>
                    )}

                    <div className="run-session-actions">
                      <button
                        type="button"
                        className="run-button"
                        disabled={
                          openingTunnelId !== null ||
                          !['running', 'queued'].includes(session.job.status) ||
                          Boolean(session.unreachable) ||
                          Boolean(session.read_error) ||
                          ['running', 'queued'].includes(
                            session.tunnel?.status ?? '',
                          )
                        }
                        onClick={() => void handleOpenTunnel(session.job.id)}
                      >
                        {openingTunnelId === session.job.id
                          ? 'Opening tunnel…'
                          : 'Open SSH tunnel'}
                      </button>
                      <button
                        type="button"
                        className="run-button"
                        disabled={
                          stoppingPolicyId !== null ||
                          !['running', 'queued'].includes(session.job.status)
                        }
                        onClick={() => void handleStopPolicy(session.job.id)}
                      >
                        {stoppingPolicyId === session.job.id
                          ? 'Stopping…'
                          : 'Stop policy server'}
                      </button>
                      <button
                        type="button"
                        className="run-button"
                        disabled={logLoading}
                        aria-expanded={logJobId === session.job.id}
                        onClick={() => void handleViewLogs(session.job.id)}
                      >
                        {logJobId === session.job.id ? 'Hide logs' : 'View logs'}
                      </button>
                    </div>
                    {logJobId === session.job.id && (
                      <div className="run-session-log">
                        {logError && <p role="alert">{logError}</p>}

                        <pre className="run-log">
                          {logLoading
                            ? 'Loading logs…'
                            : logContent || 'No log output yet.'}
                        </pre>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </section>
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
              <button type="button" className="run-button" disabled>
                Stop execution
              </button>
            </div>
          </section>

          <section className="run-panel">
            <h2>Execution logs</h2>
            <p className="run-description">
              Robot execution logs will appear here once rollout logging is
              connected.
            </p>
          </section>
        </div>

        <div className="run-controls">
          <section className="run-panel">
            <h2>Policy settings</h2>

            <label className="run-field">
              Checkpoint
              <select
                value={checkpoint}
                onChange={(event) => setCheckpoint(event.target.value)}
              >
                <option value="">
                  {checkpoints === null
                    ? 'Loading checkpoints…'
                    : 'Select a checkpoint'}
                </option>

                {checkpoints?.map((item) => (
                  <option key={item.job_id} value={item.job_id}>
                    {item.label} · step {item.step}
                  </option>
                ))}
              </select>
            </label>
            {checkpointError && <p role="alert">{checkpointError}</p>}

            <label className="run-field">
              Evaluation task
              <select
                value={task}
                onChange={(event) => setTask(event.target.value)}
              >
                <option value="pick_place">pick_place · T_sub</option>
                <option value="stack">stack · held-out</option>
                <option value="push">push · held-out</option>
                <option value="open_drawer">open_drawer · held-out</option>
                <option value="pour">pour · held-out</option>
              </select>
            </label>

            <label className="run-field">
              Task instruction
              <textarea
                rows={4}
                value={instruction}
                onChange={(event) => setInstruction(event.target.value)}
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
              disabled={!canStartPolicy}
              onClick={() => void handleStartPolicy()}
            >
              {startingPolicy ? 'Starting…' : 'Start policy server'}
            </button>

            {policyError && <p role="alert">{policyError}</p>}
          </section>

          <section className="run-panel">
            <h2>Rollout results</h2>

            <p className="run-description">
              Record whether the rollout succeeded after execution finishes.
            </p>

            <div className="run-actions">
              <button type="button" className="run-button" disabled>
                Success
              </button>
              <button type="button" className="run-button" disabled>
                Failure
              </button>
            </div>

            <p className="run-note">
              Result recording integration is pending. Results will be appended
              without deleting or overwriting existing records.
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
          Rollout records and absolute success rates for each task will be
          displayed here.
        </p>
      </section>
    </main>
  );
}

export default RunPage;
